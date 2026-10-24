import copy
from types import SimpleNamespace
import pytest
import torch
from part_ev_state_fullscale import protocol as P
from part_ev_state_fullscale.engine import create_engine
from scripts import run_part_ev_state_fullscale as M, run_part_ev_screen as old
from part_ev_fullscale.checkpoint import seed_everything, restore_rank_state, capture_rank_state
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_private_engine_no_historical_mutation():
    engine = create_engine()
    assert engine.P is P and engine.train.__globals__ is not old.train.__globals__
    assert old.P.SCHEMA == 'part_ev_context_gate_screen_v2'
    assert old.P.recipes()['ranger_short_v1']['epochs'] == 10


@pytest.mark.parametrize('index',range(7))
def test_all_seeds_backbone_parity_and_state_placement(index,monkeypatch):
    engine = create_engine()
    task = P.TASKS[index]
    model = engine.make_model(task,torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters()) == P.COUNTS[task['architecture']]
    if task['architecture'] == 'state_late4':
        assert [i for i,b in enumerate(model.mod.blocks) if b.attn.state is not None] == [4,5,6,7]
        assert all(b.attn.state.receiver.out_features == 32 for b in model.mod.blocks[4:])
    before, rng = copy.deepcopy(model.state_dict()), capture_rank_state(model)
    # Separate real gradient-equivalence test below; keep seven initialization tests small.
    monkeypatch.setattr(M,'gradient_equivalence',lambda *a:dict(passed=True))
    assert M.initial_parity(model,task,batch(),torch.device('cpu'))['passed']
    assert torch.equal(torch.get_rng_state(),rng['torch'])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_real_nonzero_state_gradient_equivalence():
    assert M.gradient_equivalence('state_late4',torch.device('cpu'))['passed']


@pytest.mark.parametrize('index',[0,3,6])
def test_ranger_resume_matches_uninterrupted_including_lr_boundary(tmp_path,index):
    engine = create_engine()
    task,recipe = P.TASKS[index],P.recipes()['ranger_full_v1']
    model = engine.make_model(task,torch.device('cpu'))
    optimizer = engine.build_optimizer(model,recipe)
    scaler = torch.amp.GradScaler('cpu',enabled=False)
    seed_everything(88)
    for step in (699997,699998):
        engine.training_step(model,optimizer,scaler,batch(),torch.device('cpu'),recipe,step)
    spec,runtime = dict(content_hash='full'),dict(cpu=True)
    engine.save_checkpoint(tmp_path/'last.pt',model,optimizer,scaler,dict(task=task,step=699999),spec,runtime)
    for step in range(699999,700004):
        engine.training_step(model,optimizer,scaler,batch(),torch.device('cpu'),recipe,step)
    expected = copy.deepcopy(model.state_dict())
    other = engine.make_model(task,torch.device('cpu'))
    opt2 = engine.build_optimizer(other,recipe)
    scale2 = torch.amp.GradScaler('cpu',enabled=False)
    state,rng = engine.load_checkpoint(tmp_path/'last.pt',other,opt2,scale2,spec,runtime,task)
    restore_rank_state(other,rng)
    for step in range(state['step'],700004):
        engine.training_step(other,opt2,scale2,batch(),torch.device('cpu'),recipe,step)
    for name,value in expected.items(): torch.testing.assert_close(other.state_dict()[name],value,rtol=0,atol=0)
    assert optimizer.step_counter == opt2.step_counter
    for a,b in zip(optimizer.slow,opt2.slow): torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match='another'):
        engine.load_checkpoint(tmp_path/'last.pt',other,opt2,scale2,dict(content_hash='short'),runtime,task)


def test_fifty_checks_first_tie_and_last_validation_completion(tmp_path,monkeypatch):
    engine = create_engine()
    task = P.TASKS[0]
    recipe = dict(P.recipes()[task['recipe']],total_steps=100,steps_per_epoch=2,validation_events_per_epoch=512)
    spec = dict(tasks=P.TASKS,recipes={task['recipe']:recipe},content_hash='fixture',inventory={})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values): return self.weight.expand(len(values[0]),-1)
    monkeypatch.setattr(P,'load_campaign',lambda _:spec)
    monkeypatch.setattr(P,'benchmark_summary',lambda *a:{})
    monkeypatch.setattr(engine,'gpu_setup',lambda _: (torch.device('cpu'),{}))
    monkeypatch.setattr(engine,'make_model',lambda *a:Tiny())
    calls = []
    def loader(*a,**kw):
        calls.append((a,kw))
        return [batch()]*110,None
    monkeypatch.setattr(engine,'build_loader',loader)
    count = []
    def validate(*a):
        count.append(True)
        # A higher result in check 40, tie at 50: selection must retain check 40.
        return dict(accuracy=.8 if len(count) in (40,50) else .7,seconds=1.,
                    macro_auc_ovo=.9,macro_auc_ovr=.9,loss=.1,events=512)
    monkeypatch.setattr(engine,'validate',validate)
    args = SimpleNamespace(output=tmp_path,task_index=0)
    engine.train(args)
    run = P.task_directory(tmp_path,0)
    done = P.read_json(run/'training_complete.json')
    assert (done['step'],done['validated_checks'],done['best_check']) == (100,50,40)
    assert len(list((run/'validation').glob('check_*.json'))) == 50
    assert len(calls) == 3  # train, validation, diagnostic: not rebuilt at each check.
    checksum = P.file_sha256(run/'best_model_val.pt')
    engine.train(args)
    assert P.file_sha256(run/'best_model_val.pt') == checksum


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires cluster GPU preflight')
def test_cuda_parity_and_gradients():
    task = P.TASKS[3]
    model = M.D.make_model(task,torch.device('cuda'))
    assert M.initial_parity(model,task,batch(),torch.device('cuda'))['passed']
