from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    BoundedFeaturewiseConditioning,
    FeedbackHBaseClassifier,
    HBaseParticleTransformer,
    PredictedPairAttentionBias,
    ResidualStructureTokenAdapter,
    build_feedback_selection,
    build_stage_e_plan,
    feedback_interface_contract,
    gate_warmup_updates,
    evaluate_posthoc_feedback_control,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    FEEDBACK_RESULT_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    with_content_hash,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}


class _PairEmbed(torch.nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.pairwise_lv_dim = 4
        self.pairwise_input_dim = 0
        self.is_symmetric = True
        self.out_dim = heads
        self.remove_self_pair = False
        self.sparse_eval = (False, False)
        self.embed = torch.nn.Conv1d(4, heads, 1)

    def forward(self, v, uu=None, mask=None):
        del v, mask
        batch, _, length, _ = uu.shape
        return self.embed(uu.reshape(batch, 4, -1)).reshape(
            batch, -1, length, length
        )


class _Block(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(dimension)

    def forward(self, x, padding_mask=None, attn_mask=None):
        if attn_mask is not None:
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


class _ClassBlock(torch.nn.Module):
    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dimension, heads, dropout=0, batch_first=True
        )

    def forward(self, x, x_cls=None, padding_mask=None):
        context = torch.cat((x_cls, x), dim=1)
        prefix = torch.zeros(
            padding_mask.shape[0], 1, dtype=torch.bool, device=x.device
        )
        value, _ = self.attn(
            x_cls,
            context,
            context,
            key_padding_mask=torch.cat((prefix, padding_mask), dim=1),
            need_weights=False,
        )
        return x_cls + value


class _Transformer(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        dimension, heads = 16, int(config["num_heads"])
        self.pair_extra_dim = int(config.get("pair_extra_dim", 0))
        self.use_amp = False
        self.pair_embed = _PairEmbed(heads)
        self.input = torch.nn.Linear(17, dimension)
        self.blocks = torch.nn.ModuleList(
            [_Block(dimension, heads) for _ in range(8)]
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dimension))
        self.cls_blocks = torch.nn.ModuleList(
            [_ClassBlock(dimension, heads) for _ in range(2)]
        )
        self.norm = torch.nn.LayerNorm(dimension)
        self.fc = torch.nn.Linear(dimension, 10)

    def embed(self, features):
        return self.input(features.transpose(1, 2))

    def forward(self, features, v=None, mask=None, uu=None):
        x = self.embed(features)
        padding = ~mask[:, 0].bool()
        bias = self.pair_embed(v, uu=uu, mask=mask)
        for block in self.blocks:
            x = block(x, padding_mask=padding, attn_mask=bias)
        cls = self.cls_token.expand(x.shape[0], 1, -1)
        for block in self.cls_blocks:
            cls = block(x, x_cls=cls, padding_mask=padding)
        return self.fc(self.norm(cls).squeeze(1))


def _weaver():
    def pairwise(xi, xj, num_outputs=4):
        base = xi[:, :1] + xj[:, :1]
        return torch.cat([base + index for index in range(num_outputs)], dim=1)

    return SimpleNamespace(
        ParticleTransformer=lambda **config: _Transformer(**config),
        pairwise_lv_fts=pairwise,
    )


def _batch():
    torch.manual_seed(37)
    mask = torch.tensor(
        [[[True, True, True, False]], [[True, True, False, False]]]
    )
    return {
        "points": torch.zeros(2, 2, 4),
        "features": torch.randn(2, 17, 4).masked_fill(~mask, 0),
        "lorentz_vectors": torch.randn(2, 4, 4).masked_fill(~mask, 0),
        "mask": mask,
    }


def test_feedback_contract_and_exact_warmup_integer_rule():
    contract = feedback_interface_contract()
    assert contract["token"]["live_particle_sequence_expanded"] is False
    assert contract["pair"]["offline_pair_feedback_allowed"] is False
    assert gate_warmup_updates(1) == 1
    assert gate_warmup_updates(2) == 1
    assert gate_warmup_updates(20) == 1
    assert gate_warmup_updates(21) == 2
    with pytest.raises(ValueError):
        gate_warmup_updates(0)


def test_token_and_film_zero_initialization_are_exact_hbase_noops():
    batch = _batch()
    for interface in ("FB_TOKEN", "FB_FILM"):
        torch.manual_seed(41)
        classifier = HBaseParticleTransformer(weaver_module=_weaver()).eval()
        expected = classifier(**batch)
        model = FeedbackHBaseClassifier(
            classifier,
            target_id="T_OFFLINE_TRACK_32",
            interface=interface,
            particle_dimension=16,
        ).eval()
        actual, prediction = model.forward_with_feedback(**batch)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert prediction["value"].shape == (2, 32)
    token = ResidualStructureTokenAdapter(3, particle_dimension=16)
    assert float(token.gamma) == 0.0
    assert token.structure_tokens({"value": torch.zeros(2, 3)}).shape == (
        2,
        4,
        128,
    )
    film = BoundedFeaturewiseConditioning(3, particle_dimension=16)
    scale, shift = film.parameters_for({"value": torch.randn(5, 3)})
    torch.testing.assert_close(scale, torch.ones_like(scale), atol=0, rtol=0)
    torch.testing.assert_close(shift, torch.zeros_like(shift), atol=0, rtol=0)
    with torch.no_grad():
        film.projection.weight.fill_(100)
    scale, shift = film.parameters_for({"value": torch.ones(5, 3)})
    assert bool(((scale >= 0.9) & (scale <= 1.1)).all())
    assert bool(((shift >= -0.1) & (shift <= 0.1)).all())


def test_pair_bias_is_warmup_zero_then_symmetric_masked_and_diagonal_zero():
    module = PredictedPairAttentionBias(
        input_dimension=16,
        pair_dimension=8,
        attention_heads=4,
        symmetric=True,
    )
    states = torch.randn(2, 4, 16)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    with torch.no_grad():
        module.raw_alpha.fill_(1)
    module.set_update(1, 20)
    warmup, _ = module(states, mask)
    torch.testing.assert_close(warmup, torch.zeros_like(warmup), atol=0, rtol=0)
    module.set_update(2, 20)
    active, prediction = module(states, mask)
    torch.testing.assert_close(active, active.transpose(2, 3))
    diagonal = active.diagonal(dim1=2, dim2=3)
    torch.testing.assert_close(diagonal, torch.zeros_like(diagonal), atol=0, rtol=0)
    assert bool((active[0, :, 3] == 0).all())
    assert prediction["value"].shape == (2, 4, 4, 8)


def test_detached_feedback_blocks_consumer_gradient_but_target_path_survives():
    model = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_TOKEN",
        gradient_path="DETACHED",
        particle_dimension=16,
    )
    with torch.no_grad():
        model.consumer.raw_gamma.fill_(0.5)
    logits, prediction = model.forward_with_feedback(**_batch())
    logits.sum().backward(retain_graph=True)
    assert all(
        parameter.grad is None or bool((parameter.grad == 0).all())
        for parameter in model.global_predictor.parameters()
    )
    model.zero_grad(set_to_none=True)
    prediction["value"].sum().backward()
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.global_predictor.parameters()
    )
    with pytest.raises(ValueError, match="oracle"):
        model.forward_with_feedback(
            **_batch(), oracle_feedback={"value": torch.zeros(2, 32)}
        )


def test_unrestricted_feedback_is_direct_token_and_exact_capacity_matched():
    semantic = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_TOKEN",
        particle_dimension=16,
    )
    unrestricted = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_TOKEN",
        particle_dimension=16,
        control="UNRESTRICTED",
    )
    semantic_count = sum(value.numel() for value in semantic.head_parameters())
    unrestricted_count = sum(
        value.numel() for value in unrestricted.head_parameters()
    )
    assert unrestricted_count == semantic_count
    assert unrestricted.capacity_ledger == {
        "contract": "hosd_unrestricted_feedback_capacity_ledger_v1",
        "reference_trainable_parameters": semantic_count,
        "unrestricted_pre_padding_trainable_parameters": (
            semantic_count
            - unrestricted.capacity_padding.count
        ),
        "inert_trainable_padding_parameters": unrestricted.capacity_padding.count,
        "matched_trainable_parameters": semantic_count,
    }
    logits, prediction = unrestricted.forward_with_feedback(**_batch())
    assert logits.shape == (2, 10)
    assert prediction["tokens"].shape == (2, 4, 128)
    plan = build_stage_e_plan(
        single_family_selection=_single_family_lock(),
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    rows = [row for row in plan["control_rows"] if row["control"] == "UNRESTRICTED"]
    assert rows and all(row["auxiliary_weight"] == 0.0 for row in rows)


def test_unrestricted_film_is_direct_latent_and_exact_capacity_matched():
    semantic = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_FILM",
        particle_dimension=16,
    )
    unrestricted = FeedbackHBaseClassifier(
        HBaseParticleTransformer(weaver_module=_weaver()),
        target_id="T_OFFLINE_TRACK_32",
        interface="FB_FILM",
        particle_dimension=16,
        control="UNRESTRICTED",
    )
    assert sum(value.numel() for value in unrestricted.head_parameters()) == sum(
        value.numel() for value in semantic.head_parameters()
    )
    assert unrestricted.capacity_ledger["inert_trainable_padding_parameters"] >= 0
    logits, prediction = unrestricted.forward_with_feedback(**_batch())
    assert logits.shape == (2, 10)
    assert prediction["tokens"].shape == (2, 4, 128)
    assert prediction["value"].shape == (2, 32)


def test_wrong_event_posthoc_contract_is_deployable_but_oracle_is_not(
    tmp_path, monkeypatch
):
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        stage_e_training,
    )

    model = torch.nn.Linear(2, 2)
    model.allow_oracle = False
    checkpoint = tmp_path / "best_model_val.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    checkpoint_sha = __import__("hashlib").sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        stage_e_training,
        "evaluate_auxiliary",
        lambda *args, **kwargs: {
            "classification_metrics": {"balanced_accuracy": 0.1},
            "auxiliary_loss": 1.0,
            "event_count": 2,
        },
    )
    monkeypatch.setattr(
        stage_e_training,
        "feedback_model_flop_ledger",
        lambda value: {"deployed_total_flops": 10},
    )
    source_row = {
        "row_id": "source",
        "encoder_component_seed": 11,
        "feedback_component_seed": 12,
    }
    source_completion = {
        "row_id": "source",
        "checkpoint_sha256": checkpoint_sha,
        "stage_e_plan_sha256": "a" * 64,
        "source": SOURCE,
        "content_hash": "b" * 64,
        "selected_epoch": 1,
        "selected_val_stop": {"balanced_accuracy": 0.1},
    }

    def run(control):
        return evaluate_posthoc_feedback_control(
            source_model=model,
            design_select_loader=[],
            output_dir=tmp_path / control,
            control_row={
                "row_id": control,
                "target_id": "T_OFFLINE_TRACK_32",
                "parameterization": "ABS",
                "auxiliary_weight": 0.3,
                "row_kind": "CONTROL",
                "interface": "FB_TOKEN",
                "gradient_path": "END_TO_END",
                "control": control,
                "pipeline_seed": 101,
            },
            source_row=source_row,
            component_group_ids=("track",),
            source_checkpoint_path=checkpoint,
            source_completion=source_completion,
            stage_e_plan_sha256="a" * 64,
            campaign_spec_sha256="c" * 64,
            lineage_hashes={"loader": "d" * 64},
            source=SOURCE,
            device="cpu",
        )

    shuffled = run("SHUFFLED_PREDICTION")
    assert shuffled["deployable"] is True
    assert shuffled["schema_version"] == 3
    oracle = run("ORACLE_SUB")
    assert oracle["deployable"] is False


def _single_family_lock():
    target_ids = (
        "T_OFFLINE_JET_10",
        "T_OFFLINE_COMPOSITION_16",
        "T_OFFLINE_TRACK_32",
        "T_OFFLINE_DENSITY_22",
        "T_OFFLINE_CA_TREE_26",
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    )
    selected = {target_id: f"row-{index}" for index, target_id in enumerate(target_ids)}
    definitions = {
        target_id: {
            "target_id": target_id,
            "parameterization": "ABS",
            "auxiliary_weight": 0.3,
            "head_type": "pair" if target_id.startswith("T_HLT_") else "global",
        }
        for target_id in target_ids
    }
    return with_content_hash(
        {
            "contract": SINGLE_FAMILY_SELECTION_CONTRACT,
            "schema_version": 2,
            "source": SOURCE,
            "stage_d_plan_sha256": "d" * 64,
            "phase_lock_sha256": "e" * 64,
            "complete_result_hashes": {},
            "selected_row_by_target": selected,
            "selected_definition_by_target": definitions,
            "cross_family_order": [
                {"ordinal": index, "target_id": target_id, "row_id": selected[target_id]}
                for index, target_id in enumerate(target_ids)
            ],
            "global_winner_row_id": selected[target_ids[0]],
            "negative_results_permitted": True,
            "performance_based_termination": False,
        }
    )


def test_stage_e_exact_bound_and_all_negative_selection_complete():
    plan = build_stage_e_plan(
        single_family_selection=_single_family_lock(),
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    assert plan["row_count"] == 44
    assert len(plan["scientific_rows"]) == 16
    assert len(plan["control_rows"]) == 28
    assert plan["promoted_global_targets"] == [
        "T_OFFLINE_JET_10",
        "T_OFFLINE_COMPOSITION_16",
    ]
    labels = np.tile(np.arange(10), 3)
    logits = np.zeros((30, 10), dtype=np.float64)
    metrics = evaluate_classification(logits, labels, split="design_select")
    results = [
        with_content_hash(
            {
                "contract": FEEDBACK_RESULT_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                **row,
                "stage_e_plan_sha256": plan["content_hash"],
                "design_select": {"classification_metrics": metrics},
                "deployed_analytical_flops": 10.0,
                "deployed_parameter_count": 100,
                "training_gpu_hours": 1.0,
                "deployable": bool(row["deployable"]),
            }
        )
        for row in plan["all_rows"]
    ]
    with pytest.raises(ValueError, match="complete"):
        build_feedback_selection(
            stage_e_plan=plan, results=results[:-1], source=SOURCE
        )
    lock = build_feedback_selection(
        stage_e_plan=plan, results=results, source=SOURCE
    )
    assert lock["all_rows_completed"] is True
    assert lock["negative_gain_can_still_win"] is True
    assert lock["selected_feedback_row_id"]
