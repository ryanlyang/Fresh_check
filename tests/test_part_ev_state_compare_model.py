import copy
import pytest
import torch
from part_ev_state_compare import model as M, protocol as P
from part_ev_state_compare.engine import create_engine
from part_ev_fullscale import model as native
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state, seed_everything
from part_layerwise_screen.model import LayerwiseAttention
from scripts import run_part_ev_state_compare as driver
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('index', range(3))
def test_counts_backbone_initialization_parity_and_restoration(index, monkeypatch):
    task = P.TASKS[index]
    engine = create_engine()
    model = engine.make_model(task, torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters()) == P.COUNTS[task['architecture']]
    if index < 2:
        assert len(M.state_branches(model)) == 8
        assert all(b.receiver.out_features == 32 for b in M.state_branches(model))
        assert all(b.chunk_size == 64 for b in M.state_branches(model))
        assert all(isinstance(b.attn, LayerwiseAttention) == (index == 0) for b in model.mod.blocks)
    else:
        assert M.validate_capacity_model(model)['delta'] == 432
        assert native.MATCHED_PARTICLE_FFN_HIDDEN_DIM == 544
        seed_everything(101)
        baseline = native.build_model('part')
        for key, tensor in baseline.state_dict().items():
            if key.startswith('mod.blocks.') and any(x in key for x in ('.fc1.', '.fc2.', '.post_fc_norm.')):
                continue
            torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
        assert not any('state.' in k or 'edge_projection' in k or '.heads.' in k for k in model.state_dict())
        assert all(type(b.attn) is torch.nn.MultiheadAttention for b in model.mod.blocks)
        assert abs(P.COUNTS['state_layerwise']-P.COUNTS['part_state_matched']) < 2072/2
    before, rng = copy.deepcopy(model.state_dict()), capture_rank_state(model)
    monkeypatch.setattr(driver, 'nonzero_equivalence', lambda *a: dict(passed=True))
    parity = driver.initial_parity(model, task, batch(), torch.device('cpu'))
    assert parity['passed']
    assert torch.equal(torch.get_rng_state(), rng['torch'])
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())


def test_bias_heads_are_independent_identical_and_cleaned_on_exception(monkeypatch):
    seed_everything(101)
    model = M.build_model('state_layerwise', trim=False)
    heads = model.mod.pair_embed.reference.embed.heads
    assert len(heads) == 8
    for head in heads[1:]:
        for a, b in zip(head.parameters(), heads[0].parameters()):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
        for a, b in zip(head.buffers(), heads[0].buffers()):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    def fail(*args, **kwargs):
        raise RuntimeError('intentional')
    monkeypatch.setattr(model.mod.blocks[-1], 'forward', fail)
    values, _ = create_engine().tensors(batch(), torch.device('cpu'))
    with pytest.raises(RuntimeError, match='intentional'):
        model(*values)
    assert model._state_pair_cache.normalized is None
    assert model.mod.pair_embed._trimmed_mask is None
    for block in model.mod.blocks:
        assert block.attn._bias is None and block.attn.attention._relation_stem is None


@pytest.mark.parametrize('arch', ['state_fast', 'state_layerwise'])
def test_nonzero_full_model_optimized_gradients(arch):
    seed_everything(213)
    result = M.nonzero_equivalence(arch, torch.device('cpu'))
    assert result['passed'] and result['full_model'] and result['state_tensors'] == 32


@pytest.mark.parametrize('index', range(3))
def test_ranger_resume_exact_at_schedule_boundary(tmp_path, index):
    engine, task = create_engine(), P.TASKS[index]
    recipe = P.recipes()['ranger_full_v1']
    model = engine.make_model(task, torch.device('cpu'))
    opt = engine.build_optimizer(model, recipe)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    seed_everything(88)
    for step in (699997, 699998):
        engine.training_step(model, opt, scaler, batch(), torch.device('cpu'), recipe, step)
    spec, runtime = dict(content_hash='compare'), dict(cpu=True)
    engine.save_checkpoint(tmp_path/'last.pt', model, opt, scaler, dict(task=task, step=699999), spec, runtime)
    for step in range(699999, 700004):
        engine.training_step(model, opt, scaler, batch(), torch.device('cpu'), recipe, step)
    other = engine.make_model(task, torch.device('cpu'))
    opt2 = engine.build_optimizer(other, recipe)
    scale2 = torch.amp.GradScaler('cpu', enabled=False)
    state, rng = engine.load_checkpoint(tmp_path/'last.pt', other, opt2, scale2, spec, runtime, task)
    restore_rank_state(other, rng)
    for step in range(state['step'], 700004):
        engine.training_step(other, opt2, scale2, batch(), torch.device('cpu'), recipe, step)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(other.state_dict()[name], value, rtol=0, atol=0)
    assert opt.step_counter == opt2.step_counter
    for a,b in zip(opt.slow, opt2.slow):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_reused_controls_cannot_be_trained(tmp_path):
    for index in (3,4):
        with pytest.raises(ValueError, match='never be retrained'):
            driver.main(['--output',str(tmp_path),'--phase','train','--task-index',str(index)])


def test_reused_controls_cannot_be_benchmarked(tmp_path):
    for index in (3,4):
        with pytest.raises(ValueError, match='three new architectures'):
            driver.main(['--output',str(tmp_path),'--phase','benchmark','--task-index',str(index)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable locally')
def test_cuda_layerwise_initial_parity():
    task = P.TASKS[0]
    model = create_engine().make_model(task, torch.device('cuda'))
    assert driver.initial_parity(model, task, batch(), torch.device('cuda'))['passed']
