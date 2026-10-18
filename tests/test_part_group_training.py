import copy
from types import SimpleNamespace
import pytest
import torch
from part_group_screen import protocol as P
from part_group_screen.config import TASKS
from part_group_screen.model import build_model
from part_group_screen.diagnostics import initial_parity, collect
from part_screen_common import engine as E
from part_ev_fullscale.checkpoint import seed_everything, restore_rank_state, capture_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


def engine(): return E.ScreenEngine(P,build_model,initial_parity,collect)


@pytest.mark.parametrize('task',TASKS)
def test_real_model_optimizer_resume(task,tmp_path):
    runner=engine(); device=torch.device('cpu'); recipe=P.recipes()['ranger_short_v1']
    model=runner.make_model(task,device); optimizer=build_optimizer(model,recipe)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    seed_everything(121)
    # Cross a Lookahead boundary after restart.
    for step in range(3): E.training_step(model,optimizer,scaler,batch(),device,recipe,step)
    spec,runtime={'content_hash':'test'},{'fixture':True}
    runner.save(tmp_path/'last.pt',model,optimizer,scaler,dict(task=task,step=3),spec,runtime)
    for step in range(3,7): E.training_step(model,optimizer,scaler,batch(),device,recipe,step)
    expected=copy.deepcopy(model.state_dict())
    second=runner.make_model(task,device); opt=build_optimizer(second,recipe)
    scale=torch.amp.GradScaler('cpu',enabled=False)
    state,rng=runner.load(tmp_path/'last.pt',second,opt,scale,spec,runtime,task)
    assert state['step']==3
    restore_rank_state(second,rng)
    for step in range(3,7): E.training_step(second,opt,scale,batch(),device,recipe,step)
    for key,value in second.state_dict().items(): torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='another'):
        runner.load(tmp_path/'last.pt',second,opt,scale,spec,runtime,dict(task,seed=202))


def test_diagnostics_preserve_rng_bn_and_mode():
    runner=engine(); task=TASKS[2]; device=torch.device('cpu')
    model=runner.make_model(task,device).train()
    state=capture_rank_state(model); weights=copy.deepcopy(model.state_dict())
    assert runner.parity(model,task,batch(),device)['passed']
    row=runner.diagnostics(model,batch(),device)
    assert row['gamma']==pytest.approx(.001) and len(row['occupancy'])==2
    assert model.training and model.group_branch.last_diagnostics is None
    assert torch.equal(torch.get_rng_state(),state['torch'])
    assert capture_rank_state(model)['trimmers']==state['trimmers']
    for key,value in model.state_dict().items(): assert torch.equal(value,weights[key])


def test_actual_step_function_is_reused():
    from scripts import run_part_ev_screen as original
    assert E.training_step is original.training_step
    assert E.validate is original.validate
    assert E.classification_metrics is original.classification_metrics


def test_training_orchestration_and_ties(tmp_path,monkeypatch):
    task=TASKS[0]
    recipe=dict(P.recipes()['ranger_short_v1'],total_steps=4,epochs=2,steps_per_epoch=2,validation_events_per_epoch=512)
    spec=dict(tasks=TASKS,recipes={'ranger_short_v1':recipe},content_hash='fixture',inventory={},donor={'runtime':{}})
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__();self.weight=torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values): return self.weight.expand(len(values[0]),-1)
    runner=engine()
    monkeypatch.setattr(P,'load_campaign',lambda *a,**kw:spec)
    monkeypatch.setattr(P,'require_approval',lambda *a: {})
    monkeypatch.setattr(runner,'gpu_setup',lambda s:(torch.device('cpu'),{}))
    monkeypatch.setattr(runner,'make_model',lambda *a:Tiny())
    monkeypatch.setattr(runner,'diagnostics',lambda *a: {})
    monkeypatch.setattr(E,'build_loader',lambda *a,**kw:([batch()]*20,None))
    monkeypatch.setattr(E,'validate',lambda *a:dict(accuracy=.7,seconds=1.,events=512,macro_auc_ovo=.8,macro_auc_ovr=.8,loss=.2))
    args=SimpleNamespace(output=tmp_path,task_index=0)
    runner.train(args)
    run=P.task_directory(tmp_path,0)
    done=P.read_json(run/'training_complete.json')
    assert done['step']==4 and done['validated_checks']==2 and done['best_check']==1
    checksum=P.file_sha256(run/'best_model_val.pt')
    runner.train(args)
    assert P.file_sha256(run/'best_model_val.pt')==checksum


@pytest.mark.parametrize('interrupt_after_validation', [False, True])
def test_orchestrator_restart_around_validation(tmp_path, monkeypatch, interrupt_after_validation):
    from part_ev_fullscale.checkpoint import trusted_load
    task = TASKS[0]
    recipe = dict(P.recipes()['ranger_short_v1'], total_steps=8, epochs=2,
                  steps_per_epoch=4, validation_events_per_epoch=512)
    spec = dict(tasks=TASKS, recipes={'ranger_short_v1':recipe}, content_hash='fixture',
                inventory={}, donor={'runtime':{}})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values): return self.weight.expand(len(values[0]),-1)
    runner = engine()
    monkeypatch.setattr(P,'load_campaign',lambda *a,**kw:spec)
    monkeypatch.setattr(P,'require_approval',lambda *a:{})
    monkeypatch.setattr(runner,'gpu_setup',lambda _:(torch.device('cpu'),{}))
    monkeypatch.setattr(runner,'make_model',lambda *a:Tiny())
    monkeypatch.setattr(runner,'diagnostics',lambda *a:{})
    monkeypatch.setattr(E,'build_loader',lambda *a,**kw:([batch()]*20,None))
    monkeypatch.setattr(E,'validate',lambda *a:dict(accuracy=.7,seconds=1.,events=512,macro_auc_ovo=.8,macro_auc_ovr=.8,loss=.2))
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    runner.train(SimpleNamespace(output=full,task_index=0))
    original_save = runner.save
    def interrupt(path, model, optimizer, scaler, state, *args):
        original_save(path, model, optimizer, scaler, state, *args)
        if state['step'] == 4 and state['validated_checks'] == int(interrupt_after_validation):
            raise InterruptedError('simulated terminal interruption')
    monkeypatch.setattr(runner,'save',interrupt)
    with pytest.raises(InterruptedError):
        runner.train(SimpleNamespace(output=resumed,task_index=0))
    monkeypatch.setattr(runner,'save',original_save)
    runner.train(SimpleNamespace(output=resumed,task_index=0))
    a = trusted_load(P.task_directory(full,0)/'last.pt')
    b = trusted_load(P.task_directory(resumed,0)/'last.pt')
    for key in a['model']:
        assert torch.equal(a['model'][key],b['model'][key])
    for key in ('step','validated_checks','best_check','best_accuracy','amp_skipped_steps'):
        assert a['state'][key] == b['state'][key]
    assert a['optimizer']['step_counter'] == b['optimizer']['step_counter']
    for x,y in zip(a['optimizer']['slow'],b['optimizer']['slow']):
        assert torch.equal(x,y)


def test_read_only_status_uses_donor_and_common_cutoff(tmp_path, monkeypatch, capsys):
    import json
    from scripts import run_part_group_screen as D
    control = dict(run_id='PART_RANGER',architecture='part',seed=101,recipe='ranger_short_v1')
    donor = tmp_path/'donor'
    spec = dict(content_hash='fixture',tasks=TASKS[:2],donor=dict(path=str(donor),tasks={'PART_RANGER':control}))
    monkeypatch.setattr(P,'load_campaign',lambda *a,**kw:spec)
    runs = [P.task_directory(tmp_path,i) for i in range(2)] + [donor/'runs/PART_RANGER/seed_101']
    for run, accuracies in zip(runs, ([.7,.8,.99],[.71,.81],[.69,.75,.95])):
        for i,value in enumerate(accuracies,1):
            P.write_json(run/'validation'/f'check_{i:02d}.json',dict(check=i,accuracy=value))
    before = {str(p):P.file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    D.status(tmp_path)
    result = json.loads(capsys.readouterr().out)
    assert result['common_check'] == 2
    assert result['best_through_common_delta_pp'][TASKS[0]['run_id']] == pytest.approx(5.)
    assert result['best_through_common_delta_pp'][TASKS[1]['run_id']] == pytest.approx(6.)
    assert before == {str(p):P.file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}


def test_recipe_lr_boundaries():
    r=P.recipes()['ranger_short_v1']
    assert P.learning_rate(139999,r)==.001
    assert P.learning_rate(140000,r)==pytest.approx(.001*.01**(1/15))
    assert P.learning_rate(195999,r) > P.learning_rate(196000,r)
    assert P.learning_rate(199999,r)==pytest.approx(1e-5)
    with pytest.raises(ValueError): P.learning_rate(200000,r)


def test_task_registry_never_includes_controls():
    assert len(TASKS)==4 and all(t['run_id'].startswith('PART_GROUP_') for t in TASKS)
    with pytest.raises(ValueError): P.task_directory('/unused',4)


def test_finite_evaluation_retry_and_commit_checks(tmp_path,monkeypatch):
    import numpy as np
    from part_ev_fullscale.checkpoint import atomic_torch_save
    from part_screen_common.controls import validate_predictions
    task=TASKS[0]
    recipe=dict(P.recipes()['ranger_short_v1'],total_steps=4,epochs=2,steps_per_epoch=2,
                validation_events_per_epoch=512,evaluation_population=20)
    files=[dict(name=f'input_{i}.root',num_entries=10,path=f'/fixture/{i}') for i in range(2)]
    spec=dict(tasks=[task],recipes={'ranger_short_v1':recipe},content_hash='fixture',
              inventory={'splits':{'val':{'files':files}}},donor={'runtime':{}})
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__();self.weight=torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values): return self.weight.expand(len(values[0]),-1)
    runner=engine(); model=Tiny()
    run=P.task_directory(tmp_path,0)
    run.mkdir(parents=True)
    optimizer=build_optimizer(model,recipe)
    P.write_json(run/'optimizer.json',E.optimizer_metadata(model,optimizer))
    for i in (1,2):
        P.write_json(run/'validation'/f'check_{i:02d}.json',dict(check=i,step=2*i,events=512,accuracy=.1))
    atomic_torch_save(dict(model=model.state_dict(),task=task,runtime={},campaign_sha256='fixture'),run/'best_model_val.pt')
    sha=P.file_sha256(run/'best_model_val.pt')
    P.write_json(run/'training_complete.json',dict(task=task,step=4,validated_checks=2,runtime={},
        best_check=1,best_accuracy=.1,best_checkpoint_sha256=sha,campaign_sha256='fixture'))
    monkeypatch.setattr(P,'load_campaign',lambda *a,**kw:spec)
    monkeypatch.setattr(P,'require_approval',lambda *a: {})
    monkeypatch.setattr(runner,'gpu_setup',lambda _: (torch.device('cpu'),{}))
    monkeypatch.setattr(runner,'make_model',lambda *a:Tiny())
    values={k:torch.zeros(10,1,1) for k in ('pf_points','pf_features','pf_vectors','pf_mask')}
    calls=[]
    def loader(*args):
        calls.append(args)
        return [(values,{'_label_':torch.arange(10)},{})]
    monkeypatch.setattr(E,'finite_validation_loader',loader)
    args=SimpleNamespace(output=tmp_path,task_index=0)
    runner.evaluate(args)
    result=validate_predictions(run,task,'fixture',sha,files,population=20)
    assert result['metrics']['accuracy']==.1 and len(calls)==2
    runner.evaluate(args)
    assert len(calls)==2  # verified committed predictions need no second inference
    with (run/'full_validation/file_000.npz').open('ab') as h:h.write(b'changed')
    with pytest.raises(ValueError,match='changed'):runner.evaluate(args)


def test_group_initialization_does_not_change_cuda_rng():
    if not torch.cuda.is_available(): pytest.skip('Requires CUDA admission environment')
    from part_group_screen.branch import GroupBranch
    from part_group_screen.config import CONFIG
    torch.cuda.manual_seed_all(123)
    before=torch.cuda.get_rng_state_all()
    GroupBranch('hybrid',101,CONFIG)
    assert all(torch.equal(a,b) for a,b in zip(before,torch.cuda.get_rng_state_all()))
