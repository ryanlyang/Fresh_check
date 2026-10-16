import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import run_part_ev_screen as D
from part_ev_screen import protocol as P
from part_ev_fullscale.checkpoint import seed_everything, restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads


def batch():
    return (dict(zip(('pf_points','pf_features','pf_vectors','pf_mask'), inputs(batch=2, particles=7))),
            {'_label_': torch.tensor([2,3])}, {})


@pytest.mark.parametrize('recipe_name', ['ranger_short_v1','adamw_short_v1'])
def test_actual_model_optimizer_resume_matches_uninterrupted(tmp_path, recipe_name):
    task = dict(architecture='part_ev_context_relation_gate',seed=101,recipe=recipe_name,run_id='test')
    recipe = P.recipes()[recipe_name]
    device = torch.device('cpu')
    model = D.make_model(task,device)
    optimizer = build_optimizer(model,recipe)
    scaler = torch.amp.GradScaler('cpu',enabled=False)
    seed_everything(121)
    for step in range(3): D.training_step(model,optimizer,scaler,batch(),device,recipe,step)
    spec, runtime = {'content_hash':'test'}, {'cpu_fixture':True}
    state = dict(task=task, step=3, validated_checks=0)
    D.save_checkpoint(tmp_path/'last.pt',model,optimizer,scaler,state,spec,runtime)
    for step in range(3,8): D.training_step(model,optimizer,scaler,batch(),device,recipe,step)
    expected = copy.deepcopy(model.state_dict())
    second = D.make_model(task,device)
    opt2 = build_optimizer(second,recipe)
    scale2 = torch.amp.GradScaler('cpu',enabled=False)
    restored,rng = D.load_checkpoint(tmp_path/'last.pt',second,opt2,scale2,spec,runtime,task)
    assert restored['step'] == 3
    restore_rank_state(second,rng)
    for step in range(3,8): D.training_step(second,opt2,scale2,batch(),device,recipe,step)
    for name,value in second.state_dict().items():
        torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='another'):
        D.load_checkpoint(tmp_path/'last.pt',second,opt2,scale2,spec,runtime,dict(task,seed=202))


def test_metrics_population_auc_and_zero_background():
    from sklearn.metrics import roc_auc_score
    from scipy.special import softmax
    labels = np.tile(np.arange(10),5)
    logits = np.random.default_rng(18).normal(size=(50,10)).astype('float32')
    logits[np.arange(50),labels] += 4
    metrics = D.classification_metrics(logits,labels,rejection=True)
    assert metrics['macro_auc_ovo'] == pytest.approx(roc_auc_score(labels,softmax(logits.astype('float64'),axis=1),multi_class='ovo'))
    assert metrics['macro_auc_ovr'] == pytest.approx(roc_auc_score(labels,softmax(logits.astype('float64'),axis=1),multi_class='ovr'))
    assert '0.99' in metrics['qcd_rejection']['Hqql']
    assert '0.995' in metrics['qcd_rejection']['Tbl']
    for targets in metrics['qcd_rejection'].values():
        for row in targets.values():
            if row['qcd_accepted'] == 0: assert row['rejection'] is None
    tied = D.classification_metrics(np.zeros_like(logits),labels,rejection=True)
    assert tied['qcd_rejection']['Hbb']['0.3']['rejection'] == 1.


def test_diagnostics_restore_rng_and_collect_context_gate():
    from part_ev_fullscale.checkpoint import capture_rank_state
    task = dict(architecture='part_ev_context_relation_gate',seed=101)
    model = D.make_model(task,torch.device('cpu')).train()
    state = capture_rank_state(model)
    row = D.context_diagnostic(model,batch(),torch.device('cpu'))
    assert len(row['layers']) == 8
    assert all(l['context_mean'] == 1 and l['gate_mean'] == 1 for l in row['layers'].values())
    assert torch.equal(torch.get_rng_state(),state['torch'])
    assert capture_rank_state(model)['trimmers'] == state['trimmers']
    assert model.training
    assert D.initial_parity(model, task, batch(), torch.device('cpu'))['passed']
    ev = D.make_model(dict(task, architecture='part_ev'), torch.device('cpu'))
    control = D.context_diagnostic(ev,batch(),torch.device('cpu'))
    assert len(control['layers']) == 8
    assert all(l['correction_to_native_norm'] > 0 for l in control['layers'].values())
    assert all(not block.attn._forward_hooks for block in ev.mod.blocks)


def test_training_boundaries_tie_selection_and_resume_completion(tmp_path,monkeypatch):
    # Exercise orchestration with real optimizer/checkpoint writes and small
    # budgets; production load_campaign never accepts this fixture protocol.
    task = P.TASKS[0]
    recipe = dict(P.recipes()[task['recipe']],total_steps=4,steps_per_epoch=2,validation_events_per_epoch=512)
    spec = dict(tasks=P.TASKS,recipes={task['recipe']:recipe},content_hash='fixture',inventory={})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values): return self.weight.expand(len(values[0]),-1)
    monkeypatch.setattr(P,'load_campaign',lambda _:spec)
    monkeypatch.setattr(P,'benchmark_summary',lambda *a: {})
    monkeypatch.setattr(D,'gpu_setup',lambda _: (torch.device('cpu'),{}))
    monkeypatch.setattr(D,'make_model',lambda *a:Tiny())
    monkeypatch.setattr(D,'build_loader',lambda *a,**kw: ([batch()]*20,None))
    monkeypatch.setattr(D,'validate',lambda *a: dict(accuracy=.7,seconds=1.,macro_auc_ovo=.8,macro_auc_ovr=.8,loss=.2))
    args = SimpleNamespace(output=tmp_path,task_index=0)
    D.train(args)
    run = P.task_directory(tmp_path,0)
    done = P.read_json(run/'training_complete.json')
    assert done['step'] == 4 and done['validated_checks'] == 2 and done['best_check'] == 1
    assert len(list((run/'validation').glob('*.json'))) == 2
    checksum = P.file_sha256(run/'best_model_val.pt')
    D.train(args)
    assert P.file_sha256(run/'best_model_val.pt') == checksum


def test_factorial_and_optimizer_interactions():
    from part_ev_screen.reporting import contrasts
    values = dict(PART_RANGER=.8,EV_RANGER=.81,EV_CONTEXT_RANGER=.82,EV_GATE_RANGER=.83,
                  EV_CONTEXT_GATE_RANGER=.85,PART_ADAMW=.82,EV_ADAMW=.84)
    row = contrasts(values)
    assert row['context_gate_interaction'] == pytest.approx(.01)
    assert row['optimizer_recipe_ev_interaction'] == pytest.approx(.01)
    assert row['context_vs_ev'] == pytest.approx(.01)


@pytest.mark.parametrize('memory,skipped,passes',[(100,False,True),(950,False,False),(100,True,False)])
def test_benchmark_records_validation_memory_and_rejects_bad_probes(tmp_path,monkeypatch,memory,skipped,passes):
    task = P.TASKS[0]
    limits = dict(P.BENCHMARK, warmup_steps=2,measured_steps=4,window_steps=2,full_length_steps=1)
    spec = dict(tasks=P.TASKS,recipes=P.recipes(),content_hash='probe',inventory={},benchmark=limits)
    monkeypatch.setattr(P,'load_campaign',lambda _:spec)
    monkeypatch.setattr(D,'gpu_setup',lambda _: (torch.device('cpu'),{'gpu_memory_bytes':1000}))
    monkeypatch.setattr(D,'make_model',lambda *a:torch.nn.Linear(2,2))
    monkeypatch.setattr(D,'build_loader',lambda *a,**kw: ([batch()]*20,None))
    monkeypatch.setattr(D,'initial_parity',lambda *a:dict(passed=True))
    monkeypatch.setattr(D,'validate',lambda *a:dict(accuracy=.6,seconds=2,events=128000))
    monkeypatch.setattr(D,'training_step',lambda *a,**kw:dict(loss=1.,amp_skipped=skipped))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    peaks = iter([100,100,memory])
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:next(peaks))
    args = SimpleNamespace(output=tmp_path,task_index=0)
    if passes: D.benchmark(args)
    else:
        with pytest.raises(RuntimeError,match='Benchmark'): D.benchmark(args)
    row = P.read_json(tmp_path/'benchmarks/PART_RANGER.json')
    assert row['passed'] == passes and row['validation_peak_bytes'] == memory
    assert row['measured_steps'] == 4 and len(row['windows']) == 2
    assert (tmp_path/'benchmarks/PART_RANGER_probe.pt').is_file()
    assert not (P.task_directory(tmp_path,0)/'last.pt').exists()
