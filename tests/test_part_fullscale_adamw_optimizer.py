"""AdamW grouping, actual decay, FP32 state and reproducible restart tests."""
from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from part_ev_fullscale.checkpoint import atomic_torch_save, trusted_load
from part_ev_fullscale.optimizer import (
    ADAMW_WEIGHT_DECAY_POLICY, Ranger, adamw_parameter_groups,
    build_optimizer, optimizer_metadata,
)
from part_ev_fullscale.protocol import RECIPE


def recipe():
    return {"optimizer": "AdamW", "start_lr": 1e-3, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": 1e-4,
            "weight_decay_policy": ADAMW_WEIGHT_DECAY_POLICY}


class GroupingFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 3), 0.7))
        self.weight_alias = self.weight
        self.bias = nn.Parameter(torch.full((2, 3), 0.5))
        self.bias_k = nn.Parameter(torch.full((1, 1, 3), 0.25))
        self.edge_projection = nn.Parameter(torch.full((2, 3, 4), 0.4))
        self.cls_token = nn.Parameter(torch.full((1, 1, 3), 0.6))
        self.scale = nn.Parameter(torch.full((3,), 0.8))
        self.scalar = nn.Parameter(torch.tensor(0.9))
        self.frozen_weight = nn.Parameter(torch.ones(3, 3), requires_grad=False)
        self.norm = nn.LayerNorm((2, 3))
        self.conv = nn.Conv1d(2, 3, 1)
        self.tied_weight = nn.Parameter(torch.full((2, 3), 0.3))
        self.alias_module = nn.Module()
        self.alias_module.register_parameter("bias", self.tied_weight)

    def no_weight_decay(self):
        return {"cls_token"}


def group_names(model, optimizer):
    return {group["name"]: group["parameter_names"]
            for group in optimizer_metadata(model, optimizer)["groups"]}


def set_gradients(model, step):
    for index, parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
            parameter.grad = torch.sin(values + 0.17 * step + index)


def assert_nested_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for observed, reference in zip(actual, expected):
            assert_nested_exact(observed, reference)
    else:
        assert actual == expected


def test_groups_are_complete_disjoint_deterministic_and_shared_alias_safe():
    model = GroupingFixture()
    optimizer = build_optimizer(model, recipe())
    names = group_names(model, optimizer)
    assert names["decay"] == ["weight", "edge_projection", "conv.weight"]
    assert set(names["no_decay"]) == {
        "bias", "bias_k", "cls_token", "scale", "scalar", "tied_weight",
        "norm.weight", "norm.bias", "conv.bias",
    }
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    another = build_optimizer(model, recipe())
    assert group_names(model, another) == names
    metadata = optimizer_metadata(model, optimizer)
    json.dumps(metadata, allow_nan=False)
    assert metadata["parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert metadata["groups"][0]["weight_decay"] == 1e-4
    assert metadata["groups"][1]["weight_decay"] == 0
    assert all(group["betas"] == (0.9, 0.999) for group in optimizer.param_groups)
    assert all(group["eps"] == 1e-8 for group in optimizer.param_groups)


def test_any_explicit_alias_excludes_the_shared_parameter():
    model = GroupingFixture()
    model.no_weight_decay = lambda: {"cls_token", "weight_alias"}
    names = group_names(model, build_optimizer(model, recipe()))
    assert "weight" not in names["decay"]
    assert "weight" in names["no_decay"]


def test_zero_gradient_step_applies_decoupled_decay_only_to_decay_group():
    model = GroupingFixture()
    config = {**recipe(), "start_lr": 0.1, "weight_decay": 0.2}
    optimizer = build_optimizer(model, config)
    before = {id(p): p.detach().clone() for p in model.parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            expected = before[id(parameter)] * (1 - config["start_lr"] * group["weight_decay"])
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.frozen_weight, before[id(model.frozen_weight)], rtol=0, atol=0)


def test_adamw_checkpoint_restores_fp32_moments_and_exact_next_updates(tmp_path):
    model = GroupingFixture()
    optimizer = build_optimizer(model, recipe())
    for step in range(1, 5):
        set_gradients(model, step)
        optimizer.step()
    path = tmp_path / "adamw.pt"
    atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    saved = trusted_load(path)
    restored = GroupingFixture()
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer = build_optimizer(restored, recipe())
    restored_optimizer.load_state_dict(saved["optimizer"])
    for state in restored_optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
    for step in range(5, 11):
        set_gradients(model, step)
        set_gradients(restored, step)
        optimizer.step()
        restored_optimizer.step()
        assert_nested_exact(restored.state_dict(), model.state_dict())
        assert_nested_exact(restored_optimizer.state_dict(), optimizer.state_dict())


def test_factory_preserves_historical_ranger_parameter_order_and_arithmetic():
    pytest.importorskip("weaver.utils.nn.optimizer.radam")
    model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    reference = copy.deepcopy(model)
    actual_optimizer = build_optimizer(model, RECIPE)
    expected_optimizer = Ranger(reference.parameters(), lr=RECIPE["start_lr"])
    assert type(actual_optimizer) is Ranger
    assert len(actual_optimizer.param_groups) == 1
    assert [id(p) for p in actual_optimizer.parameters()] == [id(p) for p in model.parameters()]
    for step in range(1, 14):
        set_gradients(model, step)
        set_gradients(reference, step)
        actual_optimizer.step()
        expected_optimizer.step()
        assert_nested_exact(model.state_dict(), reference.state_dict())
        assert_nested_exact(actual_optimizer.state_dict(), expected_optimizer.state_dict())
    metadata = optimizer_metadata(model, actual_optimizer)
    assert metadata["optimizer"] == "Ranger"
    assert metadata["groups"][0]["weight_decay"] == 0


@pytest.mark.parametrize("architecture", ["part", "part_ev", "part_plain", "part_plain_ev"])
def test_real_part_groups_include_ev_projections_but_not_cls_token_or_norms(architecture):
    pytest.importorskip("weaver.nn.model.ParticleTransformer")
    from part_ev_fullscale.model import build_model

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = build_model(architecture, trim=False)
        optimizer = build_optimizer(model, recipe())
        names = group_names(model, optimizer)
        decay = set(names["decay"])
        no_decay = set(names["no_decay"])
        assert "mod.cls_token" in no_decay
        parameters = dict(model.named_parameters())
        assert decay.isdisjoint(no_decay)
        assert decay | no_decay == set(parameters)
        assert all(parameters[name].ndim >= 2 for name in decay)
        assert all(not name.endswith("bias") for name in decay)
        qkv_name = "mod.blocks.0.attn.{}in_proj_weight".format(
            "reference." if architecture.endswith("_ev") else "")
        assert qkv_name in decay
        assert "mod.blocks.0.pre_attn_norm.weight" in no_decay
        assert "mod.blocks.0.c_attn" in no_decay
        edge_names = [name for name in parameters if name.endswith("edge_projection")]
        assert len(edge_names) == (8 if architecture.endswith("_ev") else 0)
        assert set(edge_names) <= decay
        assert all(parameters[name].ndim == 3 for name in edge_names)
    finally:
        torch.set_num_threads(previous_threads)


def test_unknown_optimizer_policy_non_fp32_and_unknown_exclusions_fail_closed():
    model = GroupingFixture()
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, {**recipe(), "optimizer": "SGD"})
    with pytest.raises(ValueError, match="weight_decay_policy"):
        build_optimizer(model, {**recipe(), "weight_decay_policy": "decay_everything"})
    with pytest.raises(ValueError, match="FP32"):
        build_optimizer(model.half(), recipe())
    model = GroupingFixture()
    model.no_weight_decay = lambda: {"absent_parameter"}
    with pytest.raises(ValueError, match="Unknown no_weight_decay"):
        build_optimizer(model, recipe())
    with pytest.raises(ValueError, match="trainable parameter"):
        adamw_parameter_groups(nn.Identity(), weight_decay=1e-4)
