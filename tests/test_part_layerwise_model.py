import copy
import random
import pytest
import torch
from part_layerwise_screen.model import build_model, BASE, COUNTS
from part_ev_fullscale.model import build_model as reference_model
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_screen.protocol import recipes
from scripts.run_part_layerwise_screen import initial_parity, diagnostics, ENGINE
from part_screen_common.engine import training_step
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads
from tests.test_part_ev_screen_training import batch


@pytest.mark.parametrize("architecture", BASE)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("trim", [False, True])
def test_reference_initial_parity_rng_and_dropout(architecture, training, trim):
    seed_everything(101)
    control = reference_model(BASE[architecture], trim=trim).train(training)
    rng = torch.get_rng_state().clone()
    seed_everything(101)
    model = build_model(architecture, trim=trim).train(training)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == COUNTS[architecture]
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters()) == 3752
    if trim:
        control.mod.trimmer._counter = model.mod.trimmer._counter = 5
        control.mod.trimmer.target = model.mod.trimmer.target = (.5, .5)
    values = inputs()
    with torch.no_grad():
        seed_everything(42); expected = control(*values)
        seed_everything(42); actual = model(*values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.mod.pair_embed._trimmed_mask is None
    assert all(b.attn._bias is None for b in model.mod.blocks)
    if BASE[architecture] == "part_ev":
        for a, b in zip(model.mod.blocks, control.mod.blocks):
            assert torch.equal(a.attn.attention.edge_projection, b.attn.edge_projection)
            assert a.attn.attention._relation_stem is None
    for a, b in zip(model.mod.cls_blocks, control.mod.cls_blocks):
        for key, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[key])


@pytest.mark.parametrize("architecture", BASE)
def test_heads_independent_and_trainable(architecture):
    seed_everything(101)
    model = build_model(architecture, trim=False).train()
    heads = model.mod.pair_embed.reference.embed.heads
    assert len(heads) == 8
    for key, value in heads[0].state_dict().items():
        for head in heads[1:]:
            assert torch.equal(value, head.state_dict()[key])
            assert value.data_ptr() != head.state_dict()[key].data_ptr()
    logits = model(*inputs())
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3])).backward()
    for head in heads:
        assert head[0].weight.grad is not None
        assert torch.isfinite(head[0].weight.grad).all()
        assert torch.count_nonzero(head[0].weight.grad)
    assert not torch.equal(heads[0][0].weight.grad, heads[1][0].weight.grad)
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in model.mod.pair_embed.reference.embed.prefix.parameters())
    if BASE[architecture] == "part_ev":
        assert all(torch.count_nonzero(b.attn.attention.edge_projection.grad) for b in model.mod.blocks)


def test_routing_changes_only_requested_layer_and_cleans_up():
    model = build_model("part_layerwise", trim=False).eval()
    heads = model.mod.pair_embed.reference.embed.heads
    captured = []
    def capture(module, args):
        captured.append(module._bias.detach().clone())
    handles = [b.attn.register_forward_pre_hook(capture) for b in model.mod.blocks]
    with torch.no_grad():
        model(*inputs())
        original = captured[:]
        captured.clear()
        heads[3][0].weight.add_(.1)
        model(*inputs())
    for i in range(8):
        assert torch.equal(original[i], captured[i]) == (i != 3)
    for h in handles: h.remove()
    def fail(*args): raise RuntimeError("test failure")
    handle = model.mod.blocks[2].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="test failure"): model(*inputs())
    handle.remove()
    assert all(b.attn._bias is None for b in model.mod.blocks)
    assert model.mod.pair_embed._trimmed_mask is None


@pytest.mark.parametrize("architecture", BASE)
def test_startup_parity_and_diagnostics_preserve_model_and_rng(architecture):
    task = dict(architecture=architecture, seed=101)
    model = ENGINE.make_model(task, torch.device("cpu")).train()
    weights = copy.deepcopy(model.state_dict())
    rng = capture_rank_state(model)
    assert initial_parity(model, task, batch(), torch.device("cpu"))["passed"]
    assert diagnostics(model, batch(), torch.device("cpu"))["bias_head_distance_from_layer0"] == [0.] * 8
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng["torch"])
    assert capture_rank_state(model)["trimmers"] == rng["trimmers"]
    for key, value in weights.items(): assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize("architecture", BASE)
def test_real_ranger_resume(architecture, tmp_path):
    task = dict(architecture=architecture, seed=101, recipe="ranger_short_v1")
    recipe = recipes()[task["recipe"]]
    device = torch.device("cpu")
    model = ENGINE.make_model(task, device)
    optimizer = build_optimizer(model, recipe)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    seed_everything(74)
    for step in range(3): training_step(model, optimizer, scaler, batch(), device, recipe, step)
    spec, runtime = {"content_hash": "test"}, {"cpu": True}
    ENGINE.save(tmp_path/"last.pt", model, optimizer, scaler, dict(task=task, step=3), spec, runtime)
    for step in range(3, 7): training_step(model, optimizer, scaler, batch(), device, recipe, step)
    expected = copy.deepcopy(model.state_dict())
    model2 = ENGINE.make_model(task, device)
    opt2 = build_optimizer(model2, recipe)
    scale2 = torch.amp.GradScaler("cpu", enabled=False)
    state, rng = ENGINE.load(tmp_path/"last.pt", model2, opt2, scale2, spec, runtime, task)
    restore_rank_state(model2, rng)
    for step in range(3, 7): training_step(model2, opt2, scale2, batch(), device, recipe, step)
    for key, value in expected.items(): assert torch.equal(value, model2.state_dict()[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires cluster CUDA runtime")
@pytest.mark.parametrize("architecture", BASE)
def test_cuda_fp16_step_and_initial_parity(architecture):
    task = dict(architecture=architecture, seed=101)
    device = torch.device("cuda")
    model = ENGINE.make_model(task, device)
    assert initial_parity(model, task, batch(), device)["passed"]
    recipe = recipes()["ranger_short_v1"]
    optimizer = build_optimizer(model, recipe)
    row = training_step(model, optimizer, torch.amp.GradScaler("cuda"), batch(), device, recipe, 0, check_precision=True)
    assert row["precision"]["passed"]
