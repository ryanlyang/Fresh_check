from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from teacher_logit_reco.relational_part import (
    EVALUATION_CONTRACT,
    CHECKPOINT_REGISTRATION_CONTRACT,
    DeterministicEpochSampler,
    EdgeValueAttention,
    TrainingConfig,
    build_evaluation_contract,
    build_confirmation_architecture_model,
    build_global_determinism_contract,
    build_resource_profile_contract,
    build_step6_attention_contract,
    efficient_edge_value_message,
    evaluate_logits,
    evaluate_model,
    explicit_edge_value_message,
    profile_model_resources,
    preferred_checkpoint,
    qcd_signal_rejection,
    train_relational_model,
    update_patience,
)
from teacher_logit_reco.relational_part.train import _capture_diagnostics
from teacher_logit_reco.relational_part.attention import (
    attention_allocation_diagnostics,
    capture_multihead_attention_weights,
)
from scripts.validate_relational_part_weaver_parity import (
    _assert_trimmer_state_restored,
)


def test_global_checkpoint_window_and_exact_patience() -> None:
    rows = [
        {"epoch": 1, "val_stop": {"accuracy": .80000, "cross_entropy": .50}},
        {"epoch": 2, "val_stop": {"accuracy": .80009, "cross_entropy": .60}},
        {"epoch": 3, "val_stop": {"accuracy": .80008, "cross_entropy": .40}},
    ]
    # All three lie in the window of the global maximum; CE, then epoch wins.
    assert preferred_checkpoint(rows)["epoch"] == 3
    count, reset = update_patience(rows, previous_count=4)
    assert (count, reset) == (0, True)
    rows.append(
        {"epoch": 4, "val_stop": {"accuracy": .80020, "cross_entropy": .70}}
    )
    # The new maximum excludes epochs 1/3 and makes epoch 4 globally preferred.
    assert preferred_checkpoint(rows)["epoch"] == 4
    count, reset = update_patience(rows, previous_count=7)
    assert (count, reset) == (0, True)
    rows.append(
        {"epoch": 5, "val_stop": {"accuracy": .80019, "cross_entropy": .80}}
    )
    count, reset = update_patience(rows, previous_count=0)
    assert (count, reset) == (1, False)
    with pytest.raises(FloatingPointError):
        preferred_checkpoint(
            [{"epoch": 1, "val_stop": {"accuracy": np.nan, "cross_entropy": 1.0}}]
        )


def test_float64_metrics_ece_auc_and_qcd_tie_contract() -> None:
    labels = np.repeat(np.arange(10), 3)
    logits = np.full((30, 10), -2.0)
    logits[np.arange(30), labels] = 2.0
    # Put a tied pair at the signal threshold; both must pass.
    logits[3:6, 1] = np.asarray((0.5, 0.5, -0.5))
    result = evaluate_logits(logits, labels, split="val_select")
    assert result["contract"] == EVALUATION_CONTRACT
    assert result["calculation_dtype"] == "float64"
    assert len(result["ece_15_bin_top_label"]["bins"]) == 15
    assert result["ece_15_bin_top_label"]["bins"][-1]["right_inclusive"] is True
    assert set(result["one_vs_rest_auc"]) == {
        "QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"
    }
    rejection = qcd_signal_rejection(
        logits, labels, signal_index=1, target_efficiency=.30
    )
    assert rejection["signal_pass_count"] == 2
    assert rejection["achieved_signal_efficiency"] == pytest.approx(2 / 3)
    assert rejection["pass_rule"] == "score_greater_than_or_equal_to_threshold"


def test_edge_value_efficient_reference_zero_projection_and_masking() -> None:
    torch.manual_seed(13)
    raw = torch.rand(2, 3, 4, 4)
    weights = raw / raw.sum(dim=-1, keepdim=True)
    stem = torch.randn(2, 5, 4, 4, requires_grad=True)
    projection = torch.randn(3, 2, 5, requires_grad=True)
    efficient = efficient_edge_value_message(weights, stem, projection)
    explicit = explicit_edge_value_message(weights, stem, projection)
    torch.testing.assert_close(efficient, explicit, atol=1e-6, rtol=1e-6)
    efficient.square().sum().backward()
    assert stem.grad is not None and projection.grad is not None

    reference = torch.nn.MultiheadAttention(
        6, 3, dropout=0.0, batch_first=True
    ).eval()
    wrapped = EdgeValueAttention(reference, relation_width=5).eval()
    wrapped.collect_diagnostics = True
    wrapped.edge_projection.data.zero_()
    query = torch.randn(2, 4, 6)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    relation = torch.randn(2, 5, 4, 4)
    pair_mask = valid[:, None, :, None] & valid[:, None, None, :]
    relation = relation * pair_mask
    expected, _ = reference(
        query,
        query,
        query,
        key_padding_mask=~valid,
        need_weights=False,
    )
    wrapped.bind(relation, valid)
    actual, returned = wrapped(
        query,
        query,
        query,
        key_padding_mask=~valid,
        need_weights=False,
    )
    wrapped.clear()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert returned is None
    assert wrapped.last_diagnostics["masked_query_count"] == 3
    assert wrapped.last_diagnostics["materialized_pair_value_tensor"] is False


class _CustomWeaverAttention(torch.nn.Module):
    """Small local stand-in for Weaver's newer custom Attention."""

    def __init__(self, dimension=8, heads=2):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dimension // heads
        self.dropout = 0.0
        self.in_proj = torch.nn.Linear(dimension, 3 * dimension)
        self.out_proj = torch.nn.Linear(dimension, dimension)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
    ):
        batch, query_count, dimension = query.shape
        context_count = key.shape[1]
        q, k, v = torch.nn.functional._in_projection_packed(
            query,
            key,
            value,
            self.in_proj.weight,
            self.in_proj.bias,
        )
        q = q.view(
            batch, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch, context_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch, context_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        mask = None
        if key_padding_mask is not None:
            mask = torch.zeros_like(
                key_padding_mask, dtype=query.dtype
            ).masked_fill(key_padding_mask, -torch.inf)
            mask = mask[:, None, None, :]
        if attn_mask is not None:
            mask = attn_mask if mask is None else mask + attn_mask
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, mask
        )
        output = output.transpose(1, 2).reshape(
            batch, query_count, dimension
        )
        return self.out_proj(output), None


class SequenceTrimmer(torch.nn.Module):
    """Official-Weaver-shaped eval trimmer used by capture regressions."""

    def __init__(self, warmup_steps=5):
        super().__init__()
        self.enabled = True
        self.warmup_steps = warmup_steps
        self.register_buffer(
            "_counter", torch.zeros(1, dtype=torch.long), persistent=False
        )

    def forward(self, x, v=None, mask=None, uu=None):
        if self.enabled:
            if int(self._counter.item()) < self.warmup_steps:
                self._counter.add_(1)
            else:
                maximum = int(mask.sum(dim=-1).max().item())
                x = x[..., :maximum]
                mask = mask[..., :maximum]
                if v is not None:
                    v = v[..., :maximum]
                if uu is not None:
                    uu = uu[..., :maximum, :maximum]
        return x, v, mask, uu


class _TrimmedAttentionModel(torch.nn.Module):
    def __init__(self, *, custom: bool):
        super().__init__()
        self.trimmer = SequenceTrimmer()
        self.attention = (
            _CustomWeaverAttention()
            if custom
            else torch.nn.MultiheadAttention(
                8, 2, dropout=0, batch_first=True
            )
        )
        self.last_width = None

    def forward(self, tokens, valid):
        x, _, mask, _ = self.trimmer(
            tokens.transpose(1, 2),
            mask=valid.unsqueeze(1),
        )
        x = x.transpose(1, 2)
        self.last_width = int(x.shape[1])
        kwargs = {"key_padding_mask": ~mask[:, 0]}
        if isinstance(self.attention, torch.nn.MultiheadAttention):
            kwargs["need_weights"] = False
        return self.attention(x, x, x, **kwargs)[0]


@pytest.mark.parametrize("custom", [False, True])
def test_attention_capture_disables_and_restores_active_sequence_trimming(
    custom: bool,
) -> None:
    torch.manual_seed(19)
    model = _TrimmedAttentionModel(custom=custom).eval()
    tokens = torch.randn(8, 7, 8)
    valid = torch.zeros(8, 7, dtype=torch.bool)
    for row, count in enumerate((3, 2, 4, 1, 3, 2, 4, 1)):
        valid[row, :count] = True
    for _ in range(5):
        model(tokens, valid)
    model(tokens, valid)
    assert model.last_width == 4
    counter = model.trimmer._counter.clone()
    captured = capture_multihead_attention_weights(
        model, lambda: model(tokens, valid)
    )
    assert model.last_width == 7
    assert model.trimmer.enabled is True
    assert torch.equal(model.trimmer._counter, counter)
    assert [list(value.shape[-2:]) for value in captured] == [[7, 7]]


def test_parity_attestation_rejects_trimmer_enabled_flag_drift() -> None:
    counter = torch.tensor([5], dtype=torch.long)
    trimmer = SimpleNamespace(enabled=False, _counter=counter.clone())
    with pytest.raises(AssertionError, match="enabled flag"):
        _assert_trimmer_state_restored(
            torch,
            trimmer,
            counter_before=counter,
        )


def test_custom_weaver_attention_capture_and_particle_shape_filter() -> None:
    torch.manual_seed(17)
    attention = _CustomWeaverAttention().eval()
    tokens = torch.randn(2, 4, 8)
    valid = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    captured = capture_multihead_attention_weights(
        attention,
        lambda: attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~valid,
        ),
    )
    assert [list(value.shape) for value in captured] == [[2, 2, 4, 4]]
    vectors = torch.randn(2, 4, 4)
    allocation = attention_allocation_diagnostics(
        captured,
        vectors,
        valid.unsqueeze(1),
        expected_particle_layer_count=1,
    )
    assert allocation["captured_particle_attention_layer_count"] == 1


class _MiniDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(99)
        self.features = torch.randn(20, 17, 3, generator=generator)
        self.labels = torch.arange(20) % 10

    def __len__(self):
        return 20

    def __getitem__(self, index):
        features = self.features[index]
        return {
            "points": torch.zeros(2, 3),
            "features": features,
            "lorentz_vectors": torch.ones(4, 3),
            "mask": torch.ones(1, 3, dtype=torch.bool),
            "labels": self.labels[index],
        }


class _MiniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.linear = torch.nn.Linear(17, 10)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        pooled = (features * mask).mean(-1)
        return self.linear(self.dropout(pooled))


def _loaders(seed: int):
    dataset = _MiniDataset()
    train = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=DeterministicEpochSampler(dataset, seed=seed),
    )
    validation = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    return train, validation


def _train(root: Path, model: torch.nn.Module, evaluator=evaluate_model):
    train, validation = _loaders(17)
    determinism = build_global_determinism_contract()
    return train_relational_model(
        model=model,
        train_loader=train,
        val_stop_loader=validation,
        val_select_loader=validation,
        output_dir=root,
        run_id="RPT_BASE",
        model_contract_sha256="1" * 64,
        run_registry_sha256="2" * 64,
        relation_registry_sha256="3" * 64,
        global_determinism_sha256=determinism["content_hash"],
        lineage_hashes={"split": "4" * 64, "hlt_cache": "5" * 64},
        config=TrainingConfig(
            seed=101,
            maximum_epochs=4,
            minimum_epochs=4,
            early_stop_patience=8,
            campaign_profile="miniature_test",
        ),
        evaluator=evaluator,
    )


def test_miniature_cpu_training_resume_is_exact_and_val_select_is_not_selector(
    tmp_path: Path,
) -> None:
    torch.manual_seed(41)
    uninterrupted_model = _MiniModel()
    initial = {
        name: value.clone() for name, value in uninterrupted_model.state_dict().items()
    }
    uninterrupted = _train(tmp_path / "full", uninterrupted_model)
    assert uninterrupted["contract"] == CHECKPOINT_REGISTRATION_CONTRACT
    assert uninterrupted["val_select_evaluation_count"] == 1
    assert uninterrupted["val_select_used_for_checkpoint_selection"] is False
    assert not (tmp_path / "full" / "last.pt").exists()

    class Interrupt:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated scheduler interruption")
            return evaluate_model(*args, **kwargs)

    interrupted_model = _MiniModel()
    interrupted_model.load_state_dict(initial)
    with pytest.raises(RuntimeError, match="scheduler"):
        _train(tmp_path / "resumed", interrupted_model, evaluator=Interrupt())
    assert (tmp_path / "resumed" / "last.pt").is_file()
    resumed_model = _MiniModel()
    resumed = _train(tmp_path / "resumed", resumed_model)
    assert resumed["model_state_sha256"] == uninterrupted["model_state_sha256"]
    assert resumed["selected_epoch"] == uninterrupted["selected_epoch"]
    assert not (tmp_path / "resumed" / "last.pt").exists()
    full_curves = json.loads(
        (tmp_path / "full" / "training_curves.json").read_text()
    )
    resumed_curves = json.loads(
        (tmp_path / "resumed" / "training_curves.json").read_text()
    )
    assert full_curves["rows"] == resumed_curves["rows"]


def test_population_diagnostics_accept_partial_region_batch_and_use_denominators() -> None:
    class DiagnosticModel(torch.nn.Module):
        def diagnostics(self, labels, mask):
            batch = int(labels.shape[0])
            query_count = int(mask.sum())
            fraction = 0.2 if batch == 64 else 0.8
            actual = [2] * batch
            return {
                "REGION": {
                    "actual_cluster_counts": {
                        "2": actual,
                        "_population_statistics": {
                            "2": {
                                "kind": "concatenate",
                                "values": actual,
                            }
                        },
                    },
                    "node_counts": [5] * batch,
                },
                "TRACK": {
                    "uncertainty_floor_audit": {
                        "d0": {"applicable_count": 123}
                    }
                },
                "attention_allocation": {
                    "layers": [
                        {
                            "layer": 0,
                            "per_head": [
                                {
                                    "leading_context_fraction": fraction,
                                    "_population_statistics": {
                                        "leading_context_fraction": {
                                            "kind": "ratio",
                                            "numerator": fraction * query_count,
                                            "denominator": query_count,
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                },
            }

    full_mask = torch.ones(64, 1, 3, dtype=torch.bool)
    partial_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    loader = [
        {"labels": torch.zeros(64, dtype=torch.long), "mask": full_mask},
        {"labels": torch.zeros(1, dtype=torch.long), "mask": partial_mask},
    ]
    captured = _capture_diagnostics(
        DiagnosticModel(), loader, torch.device("cpu")
    )
    assert captured["batch_count"] == 2
    assert captured["event_count"] == 65
    assert len(
        captured["values"]["REGION"]["actual_cluster_counts"]["2"]
    ) == 65
    assert len(captured["values"]["REGION"]["node_counts"]) == 65
    assert (
        captured["values"]["TRACK"]["uncertainty_floor_audit"]["d0"][
            "applicable_count"
        ]
        == 123
    )
    expected = (0.2 * 64 * 3 + 0.8) / (64 * 3 + 1)
    observed = captured["values"]["attention_allocation"]["layers"][0][
        "per_head"
    ][0]["leading_context_fraction"]
    assert observed == pytest.approx(expected)


def test_step6_contracts_bind_global_policy_and_base_controls() -> None:
    determinism = build_global_determinism_contract()
    evaluation = build_evaluation_contract(
        global_determinism_sha256=determinism["content_hash"]
    )
    resources = build_resource_profile_contract(
        global_determinism_sha256=determinism["content_hash"]
    )
    base = build_step6_attention_contract(
        run_id="RPT_BASE_EDGEVALUE",
        families=(),
        edge_value=True,
        model_contract_sha256="a" * 64,
    )
    selected = build_step6_attention_contract(
        run_id="RPT_SELECTED_LAYERWISE",
        families=("PT", "TRACK"),
        edge_value=False,
        model_contract_sha256="b" * 64,
    )
    assert evaluation["val_select_selects_checkpoint"] is False
    assert resources["parameter_count"].endswith("exact_numel")
    assert base["enabled_relations"] == ["base4"]
    assert base["base4_architecture_control"] is True
    assert selected["enabled_relations"] == ["base4", "PT", "TRACK"]
    assert selected["layerwise_bias"]["projection_count"] == 8
    assert selected["contract"] == "relational_part_step6_attention_v2"
    assert selected["schema_version"] == 2
    assert "BatchNorm1d" in selected["layerwise_bias"]["projection_tail"]

    model = _MiniModel().eval()
    batch = next(iter(_loaders(1)[1]))
    profile = profile_model_resources(
        model,
        batch,
        warmup_repetitions=0,
        measured_repetitions=1,
        model_contract_sha256="c" * 64,
    )
    assert profile["trainable_parameters"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert profile["forward_flops"] > 0
    assert profile["peak_incremental_device_memory_bytes"] is None


class _FakePairEmbed(torch.nn.Module):
    def __init__(self, input_dimension: int, heads: int):
        super().__init__()
        self.pairwise_lv_dim = 0
        self.pairwise_input_dim = input_dimension
        self.fts_embed = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dimension),
            torch.nn.Conv1d(input_dimension, 7, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(7, heads, 1),
            # Match real Weaver's post-output normalization topology.
            torch.nn.BatchNorm1d(heads),
        )


class _FakeParticleBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(dimension)

    def forward(self, x, padding_mask=None, attn_mask=None):
        if attn_mask is not None and attn_mask.ndim == 4:
            attn_mask = attn_mask.flatten(0, 1)
        update, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.norm(x + update)


class _FakeClassBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )

    def forward(self, x, x_cls=None, padding_mask=None):
        context = torch.cat([x_cls, x], dim=1)
        cls_padding = torch.zeros(
            padding_mask.shape[0],
            1,
            dtype=torch.bool,
            device=padding_mask.device,
        )
        update, _ = self.attn(
            x_cls,
            context,
            context,
            key_padding_mask=torch.cat([cls_padding, padding_mask], dim=1),
            need_weights=False,
        )
        return x_cls + update


class _FakeArchitectureTransformer(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        dimension = 16
        heads = int(config["num_heads"])
        self.pair_embed = _FakePairEmbed(
            int(config["pair_extra_dim"]), heads
        )
        self.input = torch.nn.Linear(17, dimension)
        self.blocks = torch.nn.ModuleList(
            [_FakeParticleBlock(dimension, heads) for _ in range(8)]
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dimension))
        self.cls_blocks = torch.nn.ModuleList(
            [_FakeClassBlock(dimension, heads)]
        )
        self.norm = torch.nn.LayerNorm(dimension)
        self.fc = torch.nn.Linear(dimension, 10)

    def embed(self, features):
        return self.input(features.transpose(1, 2))


def _fake_architecture_weaver():
    def transformer(**config):
        return _FakeArchitectureTransformer(**config)

    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat([base + index for index in range(num_outputs)], 1)

    return SimpleNamespace(
        ParticleTransformer=transformer, pairwise_lv_fts=pairwise
    )


def test_full_layerwise_and_edgevalue_base4_topology_zero_message_parity() -> None:
    weaver = _fake_architecture_weaver()
    torch.manual_seed(81)
    layerwise = build_confirmation_architecture_model(
        "RPT_BASE_LAYERWISE", weaver_module=weaver
    ).eval()
    torch.manual_seed(81)
    edge = build_confirmation_architecture_model(
        "RPT_BASE_EDGEVALUE", weaver_module=weaver
    ).eval()
    for attention in edge.edge_attention:
        attention.edge_projection.data.zero_()
    assert len(layerwise.layer_bias.projections) == 8
    assert len(edge.edge_attention) == 8
    assert all(
        isinstance(projection, torch.nn.Sequential)
        and isinstance(projection[0], torch.nn.Conv1d)
        and isinstance(projection[1], torch.nn.BatchNorm1d)
        for projection in layerwise.layer_bias.projections
    )
    assert not any(
        "reference_projection" in name for name, _ in edge.named_parameters()
    )
    batch = {
        "points": torch.zeros(2, 2, 4),
        "features": torch.randn(2, 17, 4),
        "lorentz_vectors": torch.randn(2, 4, 4),
        "mask": torch.tensor(
            [[[True, True, True, False]], [[True, True, False, False]]]
        ),
    }
    expected = layerwise(**batch)
    actual = edge(**batch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    diagnostics = edge.diagnostics(**batch)
    allocation = diagnostics["attention_allocation"]
    assert allocation["captured_particle_attention_layer_count"] == 8
    assert allocation["captured_attention_shapes"][-1][2:] == [1, 5]
    assert len(allocation["captured_attention_shapes"]) == 9
    assert allocation["angular_band_edges"] == [0.0, 0.05, 0.1, 0.2, 0.4]
    first_head = allocation["layers"][0]["per_head"][0]
    context_total = sum(
        first_head[name]
        for name in (
            "leading_context_fraction",
            "subleading_context_fraction",
            "soft_context_fraction",
        )
    )
    assert context_total == pytest.approx(1.0, abs=1e-6)
