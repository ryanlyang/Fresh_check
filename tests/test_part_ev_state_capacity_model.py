import copy
import pytest
import torch
from part_ev_state_capacity import model as M, protocol as P
from part_ev_state_capacity.engine import create_engine
from part_ev_fullscale import model as native
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state, seed_everything
from scripts import run_part_ev_state_capacity as driver
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('index', range(6))
def test_architecture_counts_and_isolated_initialization(index, monkeypatch):
    task = P.TASKS[index]
    engine = create_engine()
    model = engine.make_model(task, torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters()) == P.COUNTS[task['architecture']]
    if index < 3:
        assert all(b.attn.state.receiver.out_features == 32 for b in model.mod.blocks)
        assert len(model.mod.blocks) == 8
    else:
        assert M.validate_capacity_model(model)['delta'] == 40
        assert native.MATCHED_PARTICLE_FFN_HIDDEN_DIM == 544
        seed_everything(task['seed'])
        baseline = native.build_model('part')
        for key, tensor in baseline.state_dict().items():
            if key.startswith('mod.blocks.') and any(x in key for x in ('.fc1.', '.fc2.', '.post_fc_norm.')):
                continue
            torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
    before, rng = copy.deepcopy(model.state_dict()), capture_rank_state(model)
    monkeypatch.setattr(driver, 'gradient_equivalence', lambda *a: dict(passed=True))
    parity = driver.initial_parity(model, task, batch(), torch.device('cpu'))
    assert parity['passed']
    assert torch.equal(torch.get_rng_state(), rng['torch'])
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())


def test_nonzero_all_eight_optimized_gradients():
    assert driver.gradient_equivalence('state_fast', torch.device('cpu'))['passed']


@pytest.mark.parametrize('index', [0,3])
def test_ranger_resume_exact_at_schedule_boundary(tmp_path,index):
    engine, task = create_engine(), P.TASKS[index]
    recipe = P.recipes()['ranger_full_v1']
    model = engine.make_model(task, torch.device('cpu'))
    opt = engine.build_optimizer(model, recipe)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    seed_everything(88)
    for step in (699997,699998):
        engine.training_step(model,opt,scaler,batch(),torch.device('cpu'),recipe,step)
    spec, runtime = dict(content_hash='capacity'),dict(cpu=True)
    engine.save_checkpoint(tmp_path/'last.pt',model,opt,scaler,dict(task=task,step=699999),spec,runtime)
    for step in range(699999,700004):
        engine.training_step(model,opt,scaler,batch(),torch.device('cpu'),recipe,step)
    other = engine.make_model(task,torch.device('cpu'))
    opt2 = engine.build_optimizer(other,recipe)
    scale2 = torch.amp.GradScaler('cpu',enabled=False)
    state,rng = engine.load_checkpoint(tmp_path/'last.pt',other,opt2,scale2,spec,runtime,task)
    restore_rank_state(other,rng)
    for step in range(state['step'],700004):
        engine.training_step(other,opt2,scale2,batch(),torch.device('cpu'),recipe,step)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(other.state_dict()[name],value,rtol=0,atol=0)
    assert opt.step_counter == opt2.step_counter
    for a,b in zip(opt.slow,opt2.slow): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_reused_controls_cannot_be_trained(tmp_path):
    for index in range(6,10):
        with pytest.raises(ValueError,match='never be retrained'):
            driver.main(['--output',str(tmp_path),'--phase','train','--task-index',str(index)])
